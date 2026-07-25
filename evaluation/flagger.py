import os
import glob
import sys
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdMolTransforms

def screen_xyz_file(filepath, angle_tolerance=15.0):
    """
    Screens a single XYZ file for subvalency/valence errors and improper bond angles.
    """
    issues = {
        "valence_error": False,
        "improper_angles": []
    }
    
    # 1. Read XYZ file
    try:
        with open(filepath, 'r') as f:
            xyz_block = f.read()
        
        # Raw molecule from XYZ (no bonds yet)
        mol = Chem.MolFromXYZBlock(xyz_block)
        if mol is None:
            return None
    except Exception as e:
        return None

    # 2. Determine bonds and test valency via sanitization
    try:
        # Pre-condition safeguard: Force initialization of explicit valence tracking
        # to satisfy internal C++ properties before bond determination executes.
        mol.UpdatePropertyCache(strict=False)
        
        # Charges default to 0; adjust if your systems are charged
        rdDetermineBonds.DetermineBonds(mol, charge=0)
        
        # Sanitization will explicitly catch valence and aromaticity anomalies
        Chem.SanitizeMol(mol)
    except (Chem.AtomValenceException, ValueError, RuntimeError):
        issues["valence_error"] = True
    except Exception:
        # Catch-all for failed connection handling or severe geometry corruptions
        issues["valence_error"] = True

    # 3. Check for improper bond angles (only if a 3D conformer exists)
    if mol.GetNumConformers() > 0:
        conformer = mol.GetConformer()
        ring_info = mol.GetRingInfo()  # Extract structural ring connectivity
        
        # Map atomic numbers to expected VSEPR target angles (hybridization-based)
        # Note: True hybridization properties require clean sanitization. 
        # If Sanitization failed above, we default to a standard 109.5 test for heavy atoms.
        for atom in mol.GetAtoms():
            idx_j = atom.GetIdx()
            neighbors = [n.GetIdx() for n in atom.GetNeighbors()]
            
            # We need at least 2 neighbors to measure an angle around central atom j
            if len(neighbors) < 2:
                continue
                
            # Determine expected ideal angle based on coordination/hybridization
            # A rough heuristic for common organic heavy atoms (C, N, O):
            num_neighbors = len(neighbors)
            if num_neighbors == 4:
                target_angle = 109.5  # Tetrahedral
            elif num_neighbors == 3:
                target_angle = 120.0  # Trigonal planar
            elif num_neighbors == 2:
                # Handle fallback if explicit valence evaluation fails due to prior errors
                try:
                    is_linear = atom.GetExplicitValence() == 4
                except Exception:
                    is_linear = False
                target_angle = 180.0 if is_linear else 104.5 # Linear vs Bent
            else:
                continue

            # --- RING-AWARE ADJUSTMENT ---
            # Pre-evaluate if the central atom belongs to any 3- or 4-membered rings
            atom_rings = ring_info.AtomRings()
            in_3_ring = any(len(ring) == 3 and idx_j in ring for ring in atom_rings)
            in_4_ring = any(len(ring) == 4 and idx_j in ring for ring in atom_rings)
            # print(in_3_ring, in_4_ring)
            # Check every combination of neighbor pairs around the central atom
            for i in range(len(neighbors)):
                for k in range(i + 1, len(neighbors)):
                    idx_i = neighbors[i]
                    idx_k = neighbors[k]
                    
                    # Compute 3D angle in degrees
                    angle = rdMolTransforms.GetAngleDeg(conformer, idx_i, idx_j, idx_k)
                    
                    # Dynamically adjust the target angle if BOTH neighbors and the
                    # central atom share membership in the exact same small ring structure.
                    current_target = target_angle
                    if in_3_ring and any(idx_i in ring and idx_k in ring and idx_j in ring for ring in atom_rings if len(ring) == 3):
                        current_target = 60.0   # Cyclopropane target
                    elif in_4_ring and any(idx_i in ring and idx_k in ring and idx_j in ring for ring in atom_rings if len(ring) == 4):
                        current_target = 90.0   # Cyclobutane target

                    deviation = abs(angle - current_target)
                    
                    # Flag severe angle collapses or strains
                    if deviation > angle_tolerance:
                        issues["improper_angles"].append(
                            f"Atom {atom.GetSymbol()}(idx:{idx_j}) angle ({mol.GetAtomWithIdx(idx_i).GetSymbol()}-{atom.GetSymbol()}-{mol.GetAtomWithIdx(idx_k).GetSymbol()}) is {angle:.1f}° (Expected ~{current_target}°)"
                        )

    return issues

def screen_directory(directory_path, angle_tolerance=15.0):
    """
    Loops through all .xyz files in a directory and reports global metrics.
    """
    search_path = os.path.join(directory_path, "*.xyz")
    xyz_files = glob.glob(search_path)
    
    total_files = len(xyz_files)
    subvalent_or_valence_count = 0
    improper_angle_count = 0
    clean_count = 0
    
    print(f"Found {total_files} .xyz files to process.\n" + "-"*50)
    
    for filepath in xyz_files:
        filename = os.path.basename(filepath)
        result = screen_xyz_file(filepath, angle_tolerance)
        if result is None:
            print(f"[ERROR] {filename}: Failed to parse or process completely.")
            continue
            
        has_issue = False
        print_msg = f"File: {filename}"
        
        if result["valence_error"]:
            subvalent_or_valence_count += 1
            print_msg += " -> [FLAGGED: Valence/Subvalency Issue]"
            has_issue = True
            
        if result["improper_angles"]:
            improper_angle_count += 1
            if not has_issue:
                print_msg += " -> [FLAGGED: Improper Bond Angles]"
            has_issue = True
            
        if not has_issue:
            clean_count += 1
        else:
            print(print_msg)
            # Display the first 2 problematic angles found
            for ang_issue in result["improper_angles"][:2]: 
                print(f"   - {ang_issue}")
                
    print("\n" + "="*50)
    print("SCREENING SUMMARY REPORT")
    print("="*50)
    print(f"Total Systems Analyzed:      {total_files}")
    print(f"Subvalent/Valence Failures:   {subvalent_or_valence_count}")
    print(f"Improper Angle Deviations:   {improper_angle_count}")
    print(f"Clean Systems:               {clean_count}")

# --- Execution ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python flagger.py <path_to_xyz_directory>")
        sys.exit(1)
        
    target_directory = sys.argv[1] 
    
    # Run the screening (flags angles deviating by more than 20 degrees)
    if os.path.exists(target_directory):
        screen_directory(target_directory, angle_tolerance=20.0)
    else:
        print(f"Error: Path '{target_directory}' not found.")
